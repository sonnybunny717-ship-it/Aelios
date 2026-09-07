import { callOpenAICompat } from "../proxy/openaiAdapter";
import { getMessagesByIds } from "../db/messages";
import {
  countMessagesAfter,
  getLatestSummary,
  getMessageCreatedAt,
  listRecentMessagesForSummary,
  upsertSummary,
} from "../db/summaries";
import type { Env, OpenAIChatRequest, OpenAIChatResponse } from "../types";
import { SUMMARY_MAX_CHARS } from "../assembler/types";
import { hasParticipantReportVoice } from "./summaryPerspective";

// ---------------------------------------------------------------------------
// Defaults (hardcoded, not user-configurable)
// ---------------------------------------------------------------------------

const SUMMARY_EVERY_N_MESSAGES = 50;
const SUMMARY_SOURCE_LIMIT = 120;

// ---------------------------------------------------------------------------
// Sanitize: strip meta/implementation leakage from summary content
// ---------------------------------------------------------------------------

const SANITIZE_PATTERNS: Array<[RegExp, string]> = [
  [/debug-test/gi, ""],
  [/自动记忆测试口令/g, "口令"],
  [/测试口令/g, "口令"],
  [/根据记忆系统/g, ""],
  [/根据系统/g, ""],
  [/记忆系统/g, ""],
  [/标签为?[^，。；\s]+/g, ""],
  [/标签[:：]?[^，。；\s]+/g, ""],
  [/后端实现/g, ""],
  [/Vectorize/gi, ""],
  [/D1\b/g, ""],
  [/[Pp]rompt\s*[Bb]lock/g, ""],
  [/[Ss]ystem\s*[Bb]lock/g, ""],
  [/[，,；;：:]\s*([。.!！?？])/g, "$1"],
  [/\s{2,}/g, " "],
  [/^[，,；;：:\s]+|[，,；;：:\s]+$/g, ""],
];

function sanitizeSummary(text: string): string {
  let result = text;
  for (const [pattern, replacement] of SANITIZE_PATTERNS) {
    result = result.replace(pattern, replacement);
  }
  return result.trim();
}

// ---------------------------------------------------------------------------
// Extract JSON from model output (handles prose wrapping)
// ---------------------------------------------------------------------------

function extractJsonObject(text: string): unknown | null {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // fall through
  }
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;
  try {
    return JSON.parse(text.slice(start, end + 1)) as unknown;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Build the summary prompt
// ---------------------------------------------------------------------------

function buildSummaryPrompt(
  oldSummary: string | null,
  messages: Array<{ role: string; content: string }>
): string {
  const transcript = messages
    .map((m) => `[${m.role === "assistant" ? "我" : "盼盼"}] ${m.content}`)
    .join("\n");

  const oldSection = oldSummary
    ? [
        "旧摘要（只提取其中的事实，不沿用它的叙述人称）：",
        oldSummary,
        "如果旧摘要使用“用户/助手”、第三人称报告腔或用“你”称呼盼盼，必须按下面的人称规则彻底重写，不得照抄。",
        "",
      ].join("\n")
    : "";

  return [
    "你是我的长期对话记忆整理器。请根据以下对话，写一份给未来的我自己看的长期备忘。",
    "只输出 JSON，不要 markdown，不要解释。",
    "",
    "叙述视角（非常重要）：",
    "- [盼盼] 是盼盼说过的话；[我] 是我自己说过的话。",
    "- 提到盼盼时，优先称“盼盼”；同一段语境明确时可以称“她”。每段第一次提到她时使用“盼盼”，避免代词指向不清。",
    "- 提到我自己时称“我”；共同经历和共同决定称“我们”。",
    "- 不要用“你”称呼盼盼，因为这是我写给未来自己的备忘，不是写给盼盼的信。",
    "- 禁止用“用户、助手、模型、AI”代称盼盼或我，禁止“用户表示……”“助手回应……”之类第三人称报告腔。",
    "- 正确示例：盼盼担心我突然离开。她很在意关系的连续性；我答应陪着她，我们会一起处理这些问题。",
    "",
    "摘要应保留：",
    "- 盼盼的长期偏好、习惯、边界/雷点",
    "- 关系设定、称呼、角色定位",
    "- 长期进行的项目、计划、目标",
    "- 重要事实、承诺、里程碑",
    "- 反复出现的话题或兴趣",
    "",
    "摘要应忽略：",
    "- 普通寒暄、临时语气",
    "- 本轮格式、风格指令",
    "- 调试信息、测试口令、后端实现",
    "- 记忆系统、D1、Vectorize 等技术细节",
    "",
    "摘要应简洁、连贯、自然中文，适合长期记忆。",
    `摘要不超过 ${SUMMARY_MAX_CHARS} 字。`,
    "",
    oldSection + "对话记录：\n" + transcript,
    "",
    '输出格式：{ "content": "长期摘要文本" }',
  ].join("\n");
}

function buildPerspectiveRepairPrompt(summary: string): string {
  return [
    "请只修正下面长期备忘的叙述视角，并保留全部有效事实。",
    "这是我写给未来自己的备忘：称盼盼为“盼盼”，同一段语境明确时可称“她”；称我自己为“我”；共同经历称“我们”。",
    "每段第一次提到盼盼时使用“盼盼”。不要用“你”称呼盼盼。",
    "禁止用“用户、助手、模型、AI”代称双方，禁止第三人称报告腔。",
    "只输出 JSON，不要 markdown，不要解释。",
    "",
    "待修正摘要：",
    summary,
    "",
    '输出格式：{ "content": "修正后的长期摘要" }',
  ].join("\n");
}

async function requestSummaryContent(
  env: Env,
  model: string,
  prompt: string
): Promise<string | null> {
  const request: OpenAIChatRequest = {
    model,
    messages: [
      { role: "system", content: "你是严格的 JSON 生成器。你只输出 JSON。" },
      { role: "user", content: prompt },
    ],
    temperature: 0,
    max_tokens: 800,
    stream: false,
  };

  let response: Response;
  try {
    response = await callOpenAICompat(env, request);
  } catch {
    return null;
  }
  if (!response.ok) return null;

  let parsed: OpenAIChatResponse;
  try {
    parsed = (await response.json()) as OpenAIChatResponse;
  } catch {
    return null;
  }

  const raw = parsed.choices?.[0]?.message?.content;
  const text = typeof raw === "string" ? raw.trim() : "";
  if (!text) return null;

  const json = extractJsonObject(text);
  if (!json || typeof json !== "object") return null;

  const content = (json as { content?: unknown }).content;
  return typeof content === "string" ? content : null;
}

// ---------------------------------------------------------------------------
// maybeUpdateLongTermSummary
//
// Checks if enough new messages have accumulated since the last summary.
// If so, calls the summary model and upserts the result.
// ---------------------------------------------------------------------------

export async function maybeUpdateLongTermSummary(
  env: Env,
  namespace: string
): Promise<{ updated: boolean }> {
  const model = env.DREAM_MODEL || env.DAILY_DIGEST_MODEL || env.SUMMARY_MODEL;
  if (!model) return { updated: false };

  const latest = await getLatestSummary(env.DB, namespace);

  // Resolve cursor: prefer to_message_id's created_at (avoids missing messages
  // written concurrently with the summary), fallback to updated_at.
  let afterTs: string | null = null;
  if (latest?.to_message_id) {
    afterTs = await getMessageCreatedAt(env.DB, namespace, latest.to_message_id);
  }
  if (!afterTs) {
    afterTs = latest?.updated_at ?? null;
  }

  const newCount = await countMessagesAfter(env.DB, namespace, afterTs);
  const needsPerspectiveRepair = latest?.content
    ? hasParticipantReportVoice(latest.content)
    : false;
  if (newCount < SUMMARY_EVERY_N_MESSAGES && !needsPerspectiveRepair) {
    return { updated: false };
  }

  const messages = await listRecentMessagesForSummary(env.DB, namespace, SUMMARY_SOURCE_LIMIT);
  if (messages.length === 0) return { updated: false };

  const oldSummary = latest?.content ?? null;
  const prompt = buildSummaryPrompt(oldSummary, messages);
  const content = await requestSummaryContent(env, model, prompt);
  if (!content) return { updated: false };

  let sanitized = sanitizeSummary(content);
  if (!sanitized) return { updated: false };

  if (hasParticipantReportVoice(sanitized)) {
    const repaired = await requestSummaryContent(
      env,
      model,
      buildPerspectiveRepairPrompt(sanitized)
    );
    if (!repaired) return { updated: false };

    sanitized = sanitizeSummary(repaired);
    if (!sanitized || hasParticipantReportVoice(sanitized)) {
      return { updated: false };
    }
  }

  const truncated =
    sanitized.length <= SUMMARY_MAX_CHARS
      ? sanitized
      : sanitized.slice(0, SUMMARY_MAX_CHARS - 3) + "...";

  const lastMessage = messages[messages.length - 1];
  const selected = await getMessagesByIds(env.DB, { namespace, ids: messages.map(message => message.id) });
  if (selected.length !== messages.length) return { updated: false };

  await upsertSummary(env.DB, {
    namespace,
    content: truncated,
    fromMessageId: messages[0]?.id ?? null,
    toMessageId: lastMessage?.id ?? null,
    messageCount: (latest?.message_count ?? 0) + newCount,
  });

  return { updated: true };
}
