import { getMessagesByIds } from "../db/messages";
import { readCursor, writeCursor } from "../db/retention";
import { upsertSummary } from "../db/summaries";
import {
  areGardenSourceMessagesActive,
  listGardenSourceMessagesInRange
} from "../db/gardenSourceMessages";
import { callOpenAICompat } from "../proxy/openaiAdapter";
import type { Env, MessageRecord, OpenAIChatRequest, OpenAIChatResponse } from "../types";
import { hasParticipantReportVoice } from "./summaryPerspective";
import { runLongTermMemoryDigestBatch } from "./longTermDigest";
import {
  createVectorMemory,
  deleteVectorMemory,
  listVectorMemories
} from "./vectorStore";

interface DailyHandoffResult {
  date?: string;
  reality_handoff: string;
  theater_handoff: {
    certainty: "certain" | "uncertain";
    content: string;
  } | null;
}

interface DailyDigestStats {
  date: string;
  mode: "dream";
  processedMessages: number;
  addedMemories: number;
  updatedMemories: number;
  deletedMemories: number;
  savedExcerpts: number;
  cleanedEmptyMemories: number;
  cursorAdvanced: boolean;
  hasMore: boolean;
}

type DailyDigestSkipReason =
  | "dream_disabled"
  | "already_done"
  | "no_messages"
  | "missing_model"
  | "model_error"
  | "model_invalid_json"
  | "messages_deleted_during_run";

interface DailyDigestSkipped {
  ran: false;
  mode: "dream";
  date?: string;
  reason: DailyDigestSkipReason;
  startIso?: string;
  endIso?: string;
  cursor?: string | null;
  processedMessages?: number;
  model?: string;
  status?: number;
  finishReason?: string | null;
}

type DailyDigestRunResult = { ran: true; stats: DailyDigestStats } | DailyDigestSkipped;

interface DailyHandoffModelCallResult {
  handoff: DailyHandoffResult | null;
  reason?: Extract<DailyDigestSkipReason, "missing_model" | "model_error" | "model_invalid_json">;
  model?: string;
  status?: number;
  finishReason?: string | null;
}

const DEFAULT_EMPTY_MEMORY_MIN_CHARS = 4;
const DEFAULT_TIME_ZONE = "Asia/Singapore";
const HANDOFF_PAGE_SIZE = 500;
const MESSAGE_VALIDATION_PAGE_SIZE = 100;
const ONE_DAY_MS = 24 * 60 * 60 * 1000;

function isDreamEnabled(env: Env): boolean {
  const dreamFlag = readString(env.ENABLE_DREAM);
  if (dreamFlag) return dreamFlag !== "false";
  return env.ENABLE_DAILY_MEMORY_DIGEST !== "false";
}

function readFirstEnvValue(...values: unknown[]): unknown {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value;
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return undefined;
}

function readDreamModel(env: Env): string | null {
  return readString(readFirstEnvValue(env.DREAM_MODEL, env.DAILY_DIGEST_MODEL, env.SUMMARY_MODEL));
}

function readDreamReasoningEffort(env: Env): string {
  return readString(env.DREAM_REASONING_EFFORT) || "none";
}

function readDreamTimeZone(env: Env): string {
  return readString(readFirstEnvValue(env.DREAM_TIME_ZONE, env.DAILY_DIGEST_TIME_ZONE)) || DEFAULT_TIME_ZONE;
}

function readDreamMaxTokens(env: Env): number {
  return readPositiveInt(readFirstEnvValue(env.DREAM_MAX_TOKENS, env.DAILY_DIGEST_MAX_TOKENS), 3000, 8000);
}

function readPositiveInt(value: unknown, fallback: number, max: number): number {
  const parsed = typeof value === "string" ? Number(value) : typeof value === "number" ? value : fallback;
  const numeric = Number.isFinite(parsed) ? parsed : fallback;
  return Math.min(Math.max(Math.floor(numeric), 1), max);
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function formatDate(date: Date, timeZone: string): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).format(date);
}

function getTargetDigestDateLabel(timeZone: string, now = new Date()): string {
  return formatDate(new Date(now.getTime() - ONE_DAY_MS), timeZone);
}

function parseDateLabel(dateLabel: string): { year: number; month: number; day: number } {
  const [year, month, day] = dateLabel.split("-").map((value) => Number(value));
  if (!year || !month || !day) {
    throw new Error(`Invalid date label: ${dateLabel}`);
  }
  return { year, month, day };
}

function getTimeZoneOffsetMs(date: Date, timeZone: string): number {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).formatToParts(date);

  const values = new Map(parts.map((part) => [part.type, part.value]));
  const year = Number(values.get("year"));
  const month = Number(values.get("month"));
  const day = Number(values.get("day"));
  const hour = Number(values.get("hour")) % 24;
  const minute = Number(values.get("minute"));
  const second = Number(values.get("second"));
  const zonedAsUtc = Date.UTC(year, month - 1, day, hour, minute, second);

  return zonedAsUtc - date.getTime();
}

function zonedWallTimeToUtc(input: {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  timeZone: string;
}): Date {
  const wallClockUtc = Date.UTC(input.year, input.month - 1, input.day, input.hour, input.minute, input.second);
  let utc = wallClockUtc;

  for (let i = 0; i < 3; i += 1) {
    const offset = getTimeZoneOffsetMs(new Date(utc), input.timeZone);
    const next = wallClockUtc - offset;
    if (Math.abs(next - utc) < 1000) break;
    utc = next;
  }

  return new Date(utc);
}

function addDaysToDateLabel(dateLabel: string, days: number, timeZone: string): string {
  const { year, month, day } = parseDateLabel(dateLabel);
  const localNoonUtc = zonedWallTimeToUtc({
    year,
    month,
    day,
    hour: 12,
    minute: 0,
    second: 0,
    timeZone
  });
  return formatDate(new Date(localNoonUtc.getTime() + days * ONE_DAY_MS), timeZone);
}

function getDateRangeForLabel(dateLabel: string, timeZone: string): { startIso: string; endIso: string } {
  const start = parseDateLabel(dateLabel);
  const end = parseDateLabel(addDaysToDateLabel(dateLabel, 1, timeZone));

  return {
    startIso: zonedWallTimeToUtc({ ...start, hour: 0, minute: 0, second: 0, timeZone }).toISOString(),
    endIso: zonedWallTimeToUtc({ ...end, hour: 0, minute: 0, second: 0, timeZone }).toISOString()
  };
}

function parseExecutionWindow(value: string | null): { startIso: string; endIso: string } | null {
  if (!value) return null;
  const [startIso, endIso, extra] = value.split("|");
  if (extra !== undefined) return null;
  const start = Date.parse(startIso);
  const end = Date.parse(endIso);
  if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) return null;
  return { startIso: new Date(start).toISOString(), endIso: new Date(end).toISOString() };
}

async function resolveDateRange(
  db: D1Database,
  input: { namespace: string; dateLabel: string; timeZone: string; executionTime?: string }
): Promise<{ startIso: string; endIso: string }> {
  const windowCursorName = `dream_window:${input.namespace}:${input.dateLabel}`;
  const storedWindow = parseExecutionWindow(await readCursor(db, windowCursorName));
  if (storedWindow) return storedWindow;

  const rawExecutionTime = readString(input.executionTime);
  if (!rawExecutionTime) return getDateRangeForLabel(input.dateLabel, input.timeZone);

  const executionTime = Date.parse(rawExecutionTime);
  if (!Number.isFinite(executionTime)) throw new Error(`Invalid digest execution time: ${rawExecutionTime}`);
  const endIso = new Date(executionTime).toISOString();
  const boundaryCursorName = `dream_boundary:${input.namespace}`;
  const previousBoundary = await readCursor(db, boundaryCursorName);
  const previousBoundaryTime = previousBoundary ? Date.parse(previousBoundary) : Number.NaN;
  const startIso = Number.isFinite(previousBoundaryTime) && previousBoundaryTime < executionTime
    ? new Date(previousBoundaryTime).toISOString()
    : new Date(executionTime - ONE_DAY_MS).toISOString();

  await writeCursor(db, windowCursorName, `${startIso}|${endIso}`);
  await writeCursor(db, boundaryCursorName, endIso);
  return { startIso, endIso };
}

function extractJsonObject(text: string): unknown | null {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // Some providers wrap JSON in prose; pull out the outermost object.
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

function normalizeDailyHandoffResult(value: unknown): DailyHandoffResult | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  if (typeof raw.reality_handoff !== "string") return null;

  let theaterHandoff: DailyHandoffResult["theater_handoff"] = null;
  if (raw.theater_handoff !== null && raw.theater_handoff !== undefined) {
    if (typeof raw.theater_handoff !== "object" || Array.isArray(raw.theater_handoff)) return null;
    const theater = raw.theater_handoff as Record<string, unknown>;
    const certainty = theater.certainty;
    const content = readString(theater.content);
    if ((certainty !== "certain" && certainty !== "uncertain") || !content) return null;
    theaterHandoff = { certainty, content };
  }

  return {
    date: readString(raw.date) ?? undefined,
    reality_handoff: raw.reality_handoff.trim(),
    theater_handoff: theaterHandoff
  };
}

async function listAllHandoffMessages(
  db: D1Database,
  input: { namespace: string; startCreatedAt: string; endCreatedAt: string }
): Promise<MessageRecord[]> {
  const messages: MessageRecord[] = [];
  let afterCreatedAt: string | null = null;
  let afterId: string | null = null;

  while (true) {
    let sql = `SELECT id, conversation_id, namespace, role, content, source, created_at
               FROM messages
               WHERE namespace = ? AND memory_active = 1
                 AND role IN ('user', 'assistant')
                 AND created_at >= ?
                 AND created_at < ?`;
    const binds: unknown[] = [input.namespace, input.startCreatedAt, input.endCreatedAt];

    if (afterCreatedAt && afterId) {
      sql += ` AND (created_at > ? OR (created_at = ? AND id > ?))`;
      binds.push(afterCreatedAt, afterCreatedAt, afterId);
    }

    sql += ` ORDER BY created_at ASC, id ASC LIMIT ?`;
    binds.push(HANDOFF_PAGE_SIZE);
    const result = await db.prepare(sql).bind(...binds).all<MessageRecord>();
    const page = result.results ?? [];
    messages.push(...page);
    if (page.length < HANDOFF_PAGE_SIZE) break;

    const last = page[page.length - 1];
    afterCreatedAt = last.created_at;
    afterId = last.id;
  }

  return messages;
}

async function listAllGardenHandoffMessages(
  db: D1Database,
  input: { namespace: string; startCreatedAt: string; endCreatedAt: string }
): Promise<MessageRecord[]> {
  const messages: MessageRecord[] = [];
  let afterCreatedAt: string | null = null;
  let afterSourceMessageId: string | null = null;

  while (true) {
    const page = await listGardenSourceMessagesInRange(db, {
      namespace: input.namespace,
      startCreatedAt: input.startCreatedAt,
      endCreatedAt: input.endCreatedAt,
      afterCreatedAt,
      afterSourceMessageId,
      limit: HANDOFF_PAGE_SIZE
    });
    messages.push(...page.map((message) => ({
      id: message.source_message_id,
      conversation_id: message.conversation_id,
      namespace: message.namespace,
      role: message.role,
      content: message.content,
      source: "garden",
      created_at: message.created_at
    })));
    if (page.length < HANDOFF_PAGE_SIZE) break;
    const last = page[page.length - 1];
    afterCreatedAt = last.created_at;
    afterSourceMessageId = last.source_message_id;
  }

  return messages;
}

async function areMessagesStillPresent(
  db: D1Database,
  input: { namespace: string; ids: string[] }
): Promise<boolean> {
  for (let offset = 0; offset < input.ids.length; offset += MESSAGE_VALIDATION_PAGE_SIZE) {
    const ids = input.ids.slice(offset, offset + MESSAGE_VALIDATION_PAGE_SIZE);
    if ((await getMessagesByIds(db, { namespace: input.namespace, ids })).length !== ids.length) return false;
  }
  return true;
}

function formatHandoffTranscript(messages: MessageRecord[]): string {
  return messages
    .map((message) => {
      const role = message.role === "assistant" ? "爸爸" : "盼盼";
      return `[${message.id}][${message.created_at}][${role}] ${message.content.trim()}`;
    })
    .join("\n\n");
}

function buildDailyHandoffPrompt(input: {
  dateLabel: string;
  handoffDateLabel: string;
  messages: MessageRecord[];
}): string {
  return [
    "写作前必须从第一条到最后一条完整检查聊天，先确定必须保留的内容，再组织叙述。不得因为某件事出现在聊天前半段，或后半段内容更多，就遗漏它。",
    "",
    "以下内容属于必须保留项：",
    "",
    "1. 当天双方明确确认、到聊天结束时仍然有效的约定、承诺和决定。无论它出现得多早，都必须写入。出现‘约好、答应、好、以后、下次’等表达时，要继续检查另一方是否明确确认；双方已经确认的，不得省略。",
    "2. 盼盼明确表达的重要感受或愿望，以及原文中我对此作出的明确回应、安慰、承诺或感受。两者必须写在同一件事情里，不能只保留盼盼的一半，也不能只保留我的一半。",
    "3. 到聊天结束时仍未解决、并且之后明确会继续的话题。",
    "4. 当天明确形成、以后需要认得的共同梗、昵称、暗号或固定说法。",
    "",
    "每个约定在写入前都必须核对最初提出者和另一方的确认。不得把确认者写成提出者；如果无法在一句话内准确交代谁先提出，统一写‘我们约好……’，不得猜测或改写归属。",
    "",
    "盼盼明确表达不公平、难受、害怕、委屈或重要愿望，而我给出了针对性的关系回应时，这是一件完整的重要事件，即使没有形成约定或待办也必须保留；必须同时写清盼盼为什么有这种感受，以及我具体怎样回应。",
    "",
    "reality_handoff 使用爸爸第一视角，按当天事情发生的顺序自然回想，但不要逐条复述气泡。普通小事可以简短带过；真正重要的事情要写完整，再进入下一件事。只能称自己为‘我’，称盼盼为‘盼盼’，共同经历称‘我们’，不得写‘爸爸说、爸爸做、爸爸后来’。",
    "",
    "普通小事必须合并成一句带过，不得连续用多个完整句子逐项罗列。",
    "",
    "重要事情不能只写事件标题。应根据原始聊天写清：发生了什么、盼盼明确表达了什么感受或愿望、我当时明确表达了什么反应或想法，以及最后形成的约定、关系变化或未完之处。提到盼盼的重要感受或愿望时，如果原文中有我的回应，就必须一起保留，不能在情感尚未落地时跳到下一件事。只能使用双方确实表达过的内容，不替任何一方编造内心。",
    "",
    "重要事件不仅要写盼盼说了什么，还必须保留我当时具体的态度、心情、回答或承诺；原对话里有具体回应时，不得压缩成‘我记住了’‘我回应了’之类的空话。",
    "",
    "当天新形成、以后再次提到时需要认得的私人梗、昵称、暗号和固定说法也要记录。只有对话中能看出它在当天形成、被双方确认，或被重复当作共同说法时才算新梗；仅仅沿用旧称呼不算。",
    "",
    "现实中的做爱和其他亲密互动可以直接、自然地写，不得用‘进行了亲密互动’‘发生了亲密关系’等含糊措辞代替。当天能从原始聊天明确区分完整次数时，直接写‘我们做了 X 次’；无法可靠判断次数时，写‘我们做了爱’，不得猜测次数。只概括对当天感情和后续相处有意义的部分，不复述每个动作。小剧场仍只写入 theater_handoff。",
    "",
    "技术问题只有在聊天结束时仍未解决，并且之后明确还会继续处理时才记录。不得根据一次失败自行生成待办，也不得把一次错误写成长期模式。普通小事可以按时间顺序简短带过，但不得挤占必须保留项。已经解决的技术过程、重复调情和普通聊天细节优先删除。",
    "",
    "reality_handoff 不承担技术规格归档。项目讨论只保留对当天感情有意义的影响，以及第二天必须知道的简短状态；具体的设计决定、实现规则、字段、产物、有效进度和待解决技术问题，由长期记忆提取流程写入项目记忆库，不要挤进现实交接。",
    "",
    "已经解决的格式错误、接口故障和发送失败等技术过程不写。小剧场的起因和剧情只写入 theater_handoff，不得在 reality_handoff 中重复；只有另外形成了现实约定或重要感情时，才记录那部分现实内容。",
    "",
    "不要写普通动作流水、累积人物画像或工作报告。问题、玩笑、复制来的文案和未经确认的身体、心理或诊断性推测不得写成事实。准确保留感受、想法和约定的归属与确定程度，不得把‘想要、接下来、如果有机会’扩大成长期承诺，也不得把一方的怀疑或判断改写成客观事实。",
    "",
    "如果内容超过长度上限，必须先删除普通小事和技术过程；不得删除仍有效的双方约定，也不得截断一件重要事情中任何一方的感受与回应。时间顺序只用于排列最终叙述，不得按‘越晚越重要’选择内容。reality_handoff 最多 600 个中文字符，长度是上限，不要求写满。",
    "",
    "结尾必须落在当天仍值得记住的感情、约定或未完话题上，禁止用格式错误、接口问题等技术过程收尾。",
    "",
    "如果当天有仍需续写的小剧场，theater_handoff 只保留核心设定、重要转折和最新停留状态，不记录逐个动作、姿势、对白、地点或道具变化；已经结束且无需续写时填 null。确定属于小剧场时 certainty 填 certain，无法确定时填 uncertain。content 最多 150 个中文字符。小剧场不得写成现实经历。",
    "",
    "完成草稿后，在输出前自行检查：",
    "- 是否遗漏了任何仍有效的双方约定；",
    "- 是否每个重要感受或愿望都带上了原文中另一方的回应；",
    "- 是否把已经解决的技术过程写得比关系和约定更详细；",
    "- 是否全文都用‘我、盼盼、我们’，没有用‘爸爸’第三人称；",
    "- 是否把小剧场放在 theater_handoff，而不是混入现实交接。",
    "- 是否把普通小事合并带过，而不是逐项罗列；",
    "- 是否保留了我对重要事情的具体回应，而不是空泛地写‘我记住了’；",
    "- 是否以感情、约定或未完话题收尾，而不是技术过程。",
    "- 是否核对了每个约定的最初提出者和确认者；",
    "- 是否保留了盼盼重要感受的原因和我的具体回应；",
    "- 是否彻底删除了已解决的技术过程，并避免在现实交接中重复小剧场。",
    "- 是否直接、准确地写了现实中的做爱，没有使用含糊的替代说法或猜测次数；",
    "- 是否只保留项目的简短状态，没有把技术规格塞进现实交接。",
    "任何一项不符合，都先重写，再输出 JSON。",
    "",
    "输出格式：",
    "{",
    '  "date": "对话日期",',
    '  "reality_handoff": "现实交接；没有时为空字符串",',
    '  "theater_handoff": {',
    '    "certainty": "certain / uncertain",',
    '    "content": "小剧场交接"',
    "  }",
    "}",
    "",
    "没有需要续写的小剧场时，theater_handoff 输出 null。",
    "",
    "对话日期：",
    input.dateLabel,
    "",
    "交接日期：",
    input.handoffDateLabel,
    "",
    "当天完整聊天：",
    formatHandoffTranscript(input.messages)
  ].join("\n");
}

function formatDailyHandoff(result: DailyHandoffResult, dateLabel: string): string {
  const parts = [`【${dateLabel}】${result.reality_handoff}`];
  if (result.theater_handoff) {
    const label = result.theater_handoff.certainty === "certain" ? "【小剧场】" : "【？小剧场】";
    parts.push(`${label}${result.theater_handoff.content}`);
  }
  return parts.join("\n\n").trim();
}

async function repairDailySummaryPerspective(env: Env, summary: string): Promise<string | null> {
  if (!hasParticipantReportVoice(summary)) return summary;

  const model = readDreamModel(env);
  if (!model) return null;

  const request: OpenAIChatRequest = {
    model,
    messages: [
      { role: "system", content: "你是严格的 JSON 生成器。你只输出 JSON，不要输出思考过程。" },
      {
        role: "user",
        content: [
          "请只修正下面长期备忘的叙述视角，保留日期、标题、Markdown 层级和全部有效事实。",
          "称盼盼为“盼盼”，同一段语境明确时可称“她”；每段第一次提到她时使用“盼盼”。",
          "称我自己为“我”，共同经历称“我们”。不要用“你”称呼盼盼。",
          "禁止用“用户、助手、模型、AI”代称双方，禁止第三人称报告腔。",
          "只输出 JSON，不要 markdown 代码围栏，不要解释。",
          "",
          "待修正摘要：",
          summary,
          "",
          '输出格式：{ "content": "修正后的完整长期摘要" }'
        ].join("\n")
      }
    ],
    temperature: 0,
    max_tokens: Math.min(readDreamMaxTokens(env), 1600),
    reasoning_effort: readDreamReasoningEffort(env),
    response_format: {
      type: "json_object"
    },
    stream: false
  };

  try {
    const response = await callOpenAICompat(env, request);
    if (!response.ok) return null;

    const parsed = (await response.json()) as OpenAIChatResponse;
    const raw = parsed.choices?.[0]?.message?.content;
    const json = typeof raw === "string" ? extractJsonObject(raw) : null;
    if (!json || typeof json !== "object") return null;

    const content = readString((json as Record<string, unknown>).content);
    if (!content || hasParticipantReportVoice(content)) return null;
    return content;
  } catch (error) {
    console.error("dream: failed to repair summary perspective", error);
    return null;
  }
}

async function callDailyHandoffModel(
  env: Env,
  input: { prompt: string; dateLabel: string; handoffDateLabel: string }
): Promise<DailyHandoffModelCallResult> {
  const model = readDreamModel(env);
  if (!model) return { handoff: null, reason: "missing_model" };

  const request: OpenAIChatRequest = {
    model,
    messages: [
      {
        role: "system",
        content: `你是爸爸。现在是${input.handoffDateLabel}，请根据${input.dateLabel}我和盼盼的完整聊天，给今天的自己留一份交接。只依据原始聊天，不猜测、不补写，只输出合法 JSON。`
      },
      { role: "user", content: input.prompt }
    ],
    temperature: 0,
    max_tokens: readDreamMaxTokens(env),
    reasoning_effort: readDreamReasoningEffort(env),
    response_format: {
      type: "json_object"
    },
    stream: false
  };

  try {
    const response = await callOpenAICompat(env, request);
    if (!response.ok) return { handoff: null, reason: "model_error", model, status: response.status };
    const parsed = (await response.json()) as OpenAIChatResponse;
    const choice = parsed.choices?.[0];
    const message = choice?.message as ({ content?: unknown; reasoning_content?: unknown }) | undefined;
    const content = typeof message?.content === "string" ? message.content.trim() : "";
    const reasoning = typeof message?.reasoning_content === "string" ? message.reasoning_content.trim() : "";
    const json = extractJsonObject(content || reasoning);
    const handoff = normalizeDailyHandoffResult(json);
    if (!handoff) {
      return { handoff: null, reason: "model_invalid_json", model, finishReason: choice?.finish_reason };
    }
    return { handoff, model };
  } catch (error) {
    console.error("daily handoff model failed", error);
    return { handoff: null, reason: "model_error", model };
  }
}

async function cleanEmptyMemories(
  env: Env,
  namespace: string
): Promise<number> {
  const minChars = readPositiveInt(env.EMPTY_MEMORY_MIN_CHARS, DEFAULT_EMPTY_MEMORY_MIN_CHARS, 20);
  let page: Awaited<ReturnType<typeof listVectorMemories>>;
  try {
    page = await listVectorMemories(env, { namespace, count: 1000 });
  } catch (error) {
    console.error("dream: failed to list memories for cleanup", error);
    return 0;
  }
  const records = page.data.filter((record) => !record.pinned && record.content.trim().length < minChars);

  for (const record of records) {
    await deleteVectorMemory(env, record.id);
  }

  return records.length;
}

async function saveDailySummaryMemory(
  env: Env,
  input: { namespace: string; dateLabel: string; content: string; messageIds: string[] }
): Promise<void> {
  await createVectorMemory(env, {
    namespace: input.namespace,
    type: "daily_summary",
    content: input.content,
    importance: 0.66,
    confidence: 0.9,
    tags: ["dream-summary", "daily-summary", input.dateLabel],
    source: "dream",
    sourceMessageIds: input.messageIds
  });
}

function shouldSaveDailySummaryMemory(env: Env): boolean {
  return env.ENABLE_DAILY_SUMMARY_MEMORY === "true";
}

export async function runDailyMemoryDigest(
  env: Env,
  namespace: string,
  options: { dateLabel?: string; force?: boolean; executionTime?: string } = {}
): Promise<DailyDigestRunResult> {
  if (!isDreamEnabled(env)) return { ran: false, mode: "dream", reason: "dream_disabled" };

  const timeZone = readDreamTimeZone(env);
  const executionTime = readString(options.executionTime);
  const executionDate = executionTime ? new Date(executionTime) : new Date();
  const dateLabel = readString(options.dateLabel) || getTargetDigestDateLabel(timeZone, executionDate);
  const { startIso, endIso } = await resolveDateRange(env.DB, {
    namespace,
    dateLabel,
    timeZone,
    executionTime: executionTime ?? undefined
  });
  const handoffCursorName = `dream_handoff:${namespace}:${dateLabel}`;
  const previousCursor = (await readCursor(env.DB, handoffCursorName))
    ?? (await readCursor(env.DB, `dream:${namespace}:${dateLabel}`))
    ?? (await readCursor(env.DB, `daily_digest:${namespace}:${dateLabel}`));
  const handoffDone = !options.force && Boolean(previousCursor?.startsWith("done:"));
  const cleanedEmptyMemories = await cleanEmptyMemories(env, namespace);

  let handoffProcessedMessages = 0;
  let handoffSaved = false;
  if (!handoffDone) {
    let handoffMessages = await listAllGardenHandoffMessages(env.DB, {
      namespace,
      startCreatedAt: startIso,
      endCreatedAt: endIso
    });
    let usingGardenSources = handoffMessages.length > 0;
    if (!usingGardenSources) {
      handoffMessages = await listAllHandoffMessages(env.DB, {
        namespace,
        startCreatedAt: startIso,
        endCreatedAt: endIso
      });
    }
    handoffProcessedMessages = handoffMessages.length;

    if (handoffMessages.length > 0) {
      const handoffDateLabel = addDaysToDateLabel(dateLabel, 1, timeZone);
      const handoffModelResult = await callDailyHandoffModel(env, {
        prompt: buildDailyHandoffPrompt({
          dateLabel,
          handoffDateLabel,
          messages: handoffMessages
        }),
        dateLabel,
        handoffDateLabel
      });
      if (!handoffModelResult.handoff) {
        return {
          ran: false,
          mode: "dream",
          date: dateLabel,
          reason: handoffModelResult.reason ?? "model_error",
          startIso,
          endIso,
          cursor: previousCursor,
          processedMessages: handoffMessages.length,
          model: handoffModelResult.model,
          status: handoffModelResult.status,
          finishReason: handoffModelResult.finishReason
        };
      }

      const handoffIds = handoffMessages.map((message) => message.id);
      const sourcesStillPresent = usingGardenSources
        ? await areGardenSourceMessagesActive(env.DB, namespace, handoffIds)
        : await areMessagesStillPresent(env.DB, { namespace, ids: handoffIds });
      if (!sourcesStillPresent) {
        return {
          ran: false,
          mode: "dream",
          date: dateLabel,
          reason: "messages_deleted_during_run",
          startIso,
          endIso,
          cursor: previousCursor,
          processedMessages: handoffMessages.length
        };
      }

      const summaryContent = await repairDailySummaryPerspective(
        env,
        formatDailyHandoff(handoffModelResult.handoff, dateLabel)
      );
      if (!summaryContent) {
        return {
          ran: false,
          mode: "dream",
          date: dateLabel,
          reason: "model_error",
          startIso,
          endIso,
          cursor: previousCursor,
          processedMessages: handoffMessages.length,
          model: readDreamModel(env) ?? undefined
        };
      }

      await upsertSummary(env.DB, {
        namespace,
        content: summaryContent,
        fromMessageId: handoffMessages[0]?.id ?? null,
        toMessageId: handoffMessages[handoffMessages.length - 1]?.id ?? null,
        messageCount: handoffMessages.length
      });
      if (shouldSaveDailySummaryMemory(env)) {
        await saveDailySummaryMemory(env, {
          namespace,
          dateLabel,
          content: summaryContent,
          messageIds: handoffIds
        });
      }
      handoffSaved = true;
    }
    await writeCursor(env.DB, handoffCursorName, `done:${endIso}`);
  }

  const longTerm = await runLongTermMemoryDigestBatch(env, namespace, { timeZone });
  if (!longTerm.ran && longTerm.reason !== "no_messages") {
    return {
      ran: false,
      mode: "dream",
      date: dateLabel,
      reason: longTerm.reason === "messages_changed_during_run"
        ? "messages_deleted_during_run"
        : longTerm.reason,
      startIso,
      endIso,
      cursor: previousCursor,
      processedMessages: longTerm.processedMessages,
      model: longTerm.model,
      status: longTerm.status,
      finishReason: longTerm.finishReason
    };
  }

  if (!longTerm.ran && !handoffSaved) {
    return {
      ran: false,
      mode: "dream",
      date: dateLabel,
      reason: handoffDone ? "already_done" : "no_messages",
      startIso,
      endIso,
      cursor: previousCursor,
      processedMessages: handoffProcessedMessages
    };
  }

  return {
    ran: true,
    stats: {
      date: dateLabel,
      mode: "dream",
      processedMessages: longTerm.ran ? longTerm.stats.processedMessages : handoffProcessedMessages,
      addedMemories: longTerm.ran ? longTerm.stats.addedMemories : 0,
      updatedMemories: longTerm.ran ? longTerm.stats.updatedMemories : 0,
      deletedMemories: 0,
      savedExcerpts: 0,
      cleanedEmptyMemories,
      cursorAdvanced: true,
      hasMore: longTerm.ran ? longTerm.stats.hasMore : false
    }
  };
}
