import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import type { Env, OpenAIChatRequest } from "../types";
import { openAiError } from "../utils/json";

export interface RelayTarget {
  baseUrl: string;
  apiKey: string;
}

function normalizeRelayBaseUrl(raw: string): string {
  const url = new URL(raw.trim());
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error("Relay URL must use http or https");
  }
  if (url.username || url.password) throw new Error("Relay URL must not contain credentials");

  url.search = "";
  url.hash = "";
  url.pathname = url.pathname
    .replace(/\/+$/, "")
    .replace(/\/chat\/completions$/i, "")
    .replace(/\/models$/i, "")
    .replace(/\/+$/, "");
  return url.toString().replace(/\/+$/, "");
}

function relayEndpoint(target: RelayTarget, kind: "models" | "chat/completions"): string {
  const base = target.baseUrl.replace(/\/+$/, "");
  return /\/v1$/i.test(base) ? `${base}/${kind}` : `${base}/v1/${kind}`;
}

export function readRelayTarget(request: Request): RelayTarget | null {
  const rawBaseUrl = request.headers.get("x-aelios-relay-base-url")?.trim() || "";
  const apiKey = request.headers.get("x-aelios-relay-api-key")?.trim() || "";
  if (!rawBaseUrl && !apiKey) return null;
  if (!rawBaseUrl || !apiKey) throw new Error("Relay URL and API key are both required");

  const baseUrl = normalizeRelayBaseUrl(rawBaseUrl);
  if (new URL(baseUrl).origin === new URL(request.url).origin) {
    throw new Error("Relay URL cannot point back to Aelios");
  }
  return { baseUrl, apiKey };
}

export function callRelayChat(target: RelayTarget, body: OpenAIChatRequest): Promise<Response> {
  const cleaned: OpenAIChatRequest = { ...body };
  delete cleaned.conversation_id;
  delete cleaned.context_epoch;
  delete cleaned.context_compaction;
  delete cleaned.post_user_instructions;
  delete cleaned.vision_mode;
  delete cleaned.cache_control;
  delete cleaned.session_id;
  return fetch(relayEndpoint(target, "chat/completions"), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "authorization": `Bearer ${target.apiKey}`,
    },
    body: JSON.stringify(cleaned),
  });
}

export async function handleRelayModels(request: Request, env: Env): Promise<Response> {
  const auth = await authenticate(request, env);
  if (!auth.ok) return openAiError("Unauthorized", 401, "authentication_error");
  const scopeError = requireScope(auth.profile, "chat:proxy");
  if (scopeError) return scopeError;

  let target: RelayTarget | null;
  try {
    target = readRelayTarget(request);
  } catch (error) {
    return openAiError(error instanceof Error ? error.message : "Invalid relay target", 400);
  }
  if (!target) return openAiError("Relay target is required", 400);

  try {
    const upstream = await fetch(relayEndpoint(target, "models"), {
      headers: { "authorization": `Bearer ${target.apiKey}` },
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json; charset=utf-8",
      },
    });
  } catch (error) {
    return openAiError(error instanceof Error ? error.message : "Failed to call relay", 502);
  }
}
