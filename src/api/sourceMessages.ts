import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import { syncGardenSourceMessages, type GardenSourceMessageEvent } from "../db/gardenSourceMessages";
import type { Env } from "../types";
import { json, openAiError } from "../utils/json";

const MAX_SYNC_EVENTS = 500;

function readRequiredString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readSourceMessageId(value: unknown): string | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isSafeInteger(number) && number > 0 ? String(number) : null;
}

function normalizeEvent(value: unknown): GardenSourceMessageEvent | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  const sourceMessageId = readSourceMessageId(raw.source_message_id);
  const conversationId = readRequiredString(raw.conversation_id);
  const turnId = readRequiredString(raw.turn_id);
  const role = raw.role;
  const createdAt = readRequiredString(raw.created_at);
  const createdDate = createdAt ? new Date(createdAt) : null;
  if (
    !sourceMessageId
    || !conversationId
    || !turnId
    || (role !== "user" && role !== "assistant")
    || !createdDate
    || Number.isNaN(createdDate.getTime())
  ) return null;

  return {
    sourceMessageId,
    conversationId,
    turnId,
    role,
    content: typeof raw.content === "string" ? raw.content : "",
    createdAt: createdDate.toISOString(),
    deliveryStatus: readRequiredString(raw.delivery_status) || "complete",
    active: raw.active !== false && raw.active !== 0,
    historical: raw.historical === true
  };
}

export async function handleSourceMessageSync(request: Request, env: Env): Promise<Response> {
  const auth = await authenticate(request, env);
  if (!auth.ok) return openAiError("Unauthorized", 401, "authentication_error");
  const scopeError = requireScope(auth.profile, "memory:write");
  if (scopeError) return scopeError;

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return openAiError("Request body must be valid JSON", 400);
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return openAiError("Request body must be a JSON object", 400);
  }

  const rawEvents = (body as { events?: unknown }).events;
  if (!Array.isArray(rawEvents) || rawEvents.length === 0 || rawEvents.length > MAX_SYNC_EVENTS) {
    return openAiError(`events must contain 1 to ${MAX_SYNC_EVENTS} items`, 400);
  }
  const events = rawEvents.map(normalizeEvent);
  if (events.some((event) => event === null)) return openAiError("events contains an invalid source message", 400);

  try {
    const result = await syncGardenSourceMessages(
      env.DB,
      auth.profile.namespace,
      events as GardenSourceMessageEvent[]
    );
    return json({ ok: true, ...result });
  } catch (error) {
    console.error("garden source message sync failed", error);
    return openAiError("source message sync failed", 503, "memory_error");
  }
}
