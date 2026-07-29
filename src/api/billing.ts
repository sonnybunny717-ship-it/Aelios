import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import type { Env } from "../types";
import { json, openAiError } from "../utils/json";

type BalanceResult = {
  configured: boolean;
  available: boolean;
  balance: number | null;
  currency: "USD";
  error?: string;
};

type BalanceResponse = {
  balances: {
    cloudflare: BalanceResult;
    openrouter: BalanceResult;
  };
  updatedAt: number;
};

const CACHE_MS = 30_000;
let cached: BalanceResponse | null = null;
let cachedAt = 0;

function unavailable(configured: boolean, error?: string): BalanceResult {
  return {
    configured,
    available: false,
    balance: null,
    currency: "USD",
    ...(error ? { error } : {}),
  };
}

async function cloudflareBalance(env: Env): Promise<BalanceResult> {
  const accountId = env.CLOUDFLARE_ACCOUNT_ID;
  const token = env.CLOUDFLARE_API_TOKEN;
  if (!accountId || !token) return unavailable(false);

  try {
    const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${encodeURIComponent(accountId)}/ai-gateway/billing/credit-balance`,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    if (!response.ok) return unavailable(true, `http_${response.status}`);
    const body = await response.json() as {
      result?: { balance?: number | string | null };
    };
    const balanceInCents = Number(body.result?.balance);
    if (!Number.isFinite(balanceInCents)) return unavailable(true, "invalid_response");
    // AI Gateway billing returns the credit balance in USD minor units.
    return {
      configured: true,
      available: true,
      balance: balanceInCents / 100,
      currency: "USD",
    };
  } catch {
    return unavailable(true, "request_failed");
  }
}

async function openRouterBalance(env: Env): Promise<BalanceResult> {
  const token = env.OPENROUTER_MANAGEMENT_KEY;
  if (!token) return unavailable(false);

  try {
    const response = await fetch("https://openrouter.ai/api/v1/credits", {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) return unavailable(true, `http_${response.status}`);
    const body = await response.json() as {
      data?: {
        total_credits?: number | string | null;
        total_usage?: number | string | null;
      };
    };
    const totalCredits = Number(body.data?.total_credits);
    const totalUsage = Number(body.data?.total_usage);
    if (!Number.isFinite(totalCredits) || !Number.isFinite(totalUsage)) {
      return unavailable(true, "invalid_response");
    }
    return {
      configured: true,
      available: true,
      balance: totalCredits - totalUsage,
      currency: "USD",
    };
  } catch {
    return unavailable(true, "request_failed");
  }
}

async function readBalances(env: Env, force: boolean): Promise<BalanceResponse> {
  const now = Date.now();
  if (!force && cached && now - cachedAt < CACHE_MS) return cached;

  const [cloudflare, openrouter] = await Promise.all([
    cloudflareBalance(env),
    openRouterBalance(env),
  ]);
  cached = { balances: { cloudflare, openrouter }, updatedAt: now };
  cachedAt = now;
  return cached;
}

export async function handleBillingBalances(request: Request, env: Env): Promise<Response> {
  const auth = await authenticate(request, env);
  if (!auth.ok) return openAiError("Unauthorized", 401, "authentication_error");

  const scopeError = requireScope(auth.profile, "billing:read");
  if (scopeError) return scopeError;

  const force = new URL(request.url).searchParams.get("refresh") === "1";
  return json(await readBalances(env, force), {
    headers: { "cache-control": "private, no-store" },
  });
}
