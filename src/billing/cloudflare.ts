import type { Env, TokenUsage } from "../types";

type GatewayIdentity = {
  accountId: string;
  gatewayId: string;
};

function gatewayIdentityFromBaseUrl(value: string | undefined): GatewayIdentity | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    const parts = url.pathname.split("/").filter(Boolean);
    if (url.hostname === "gateway.ai.cloudflare.com" && parts[0] === "v1") {
      const accountId = parts[1] || "";
      const gatewayId = parts[2] || "";
      return accountId && gatewayId ? { accountId, gatewayId } : null;
    }
    if (url.hostname === "api.cloudflare.com") {
      const accountIndex = parts.indexOf("accounts");
      const accountId = accountIndex >= 0 ? parts[accountIndex + 1] || "" : "";
      return accountId && parts.includes("ai")
        ? { accountId, gatewayId: "" }
        : null;
    }
  } catch {}
  return null;
}

function resolveGatewayIdentity(env: Env): GatewayIdentity | null {
  const parsed = gatewayIdentityFromBaseUrl(env.AI_GATEWAY_BASE_URL);
  const accountId = env.CLOUDFLARE_ACCOUNT_ID || parsed?.accountId || "";
  const gatewayId = env.AI_GATEWAY_ID || parsed?.gatewayId || "";
  return accountId && gatewayId ? { accountId, gatewayId } : null;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function getCloudflareGatewayLogCost(
  env: Env,
  logId: string | null
): Promise<number | null> {
  const identity = resolveGatewayIdentity(env);
  const token = env.CLOUDFLARE_API_TOKEN;
  if (!logId || !identity || !token) return null;

  const endpoint = [
    "https://api.cloudflare.com/client/v4/accounts",
    encodeURIComponent(identity.accountId),
    "ai-gateway/gateways",
    encodeURIComponent(identity.gatewayId),
    "logs",
    encodeURIComponent(logId),
  ].join("/");

  for (let attempt = 0; attempt < 3; attempt += 1) {
    if (attempt > 0) await wait(60 * attempt);
    try {
      const response = await fetch(endpoint, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        if (response.status === 404) continue;
        return null;
      }
      const body = await response.json() as {
        result?: { cost?: number | string | null };
      };
      const cost = Number(body.result?.cost);
      if (Number.isFinite(cost) && cost >= 0) return cost;
    } catch {
      return null;
    }
  }
  return null;
}

export async function addCloudflareCost(
  usage: TokenUsage | undefined,
  env: Env,
  logId: string | null
): Promise<TokenUsage | undefined> {
  if (!usage) return usage;
  const existingCost = Number(usage.cost);
  if (Number.isFinite(existingCost) && existingCost >= 0) {
    return { ...usage, cost: existingCost };
  }
  const cost = await getCloudflareGatewayLogCost(env, logId);
  return cost === null ? usage : { ...usage, cost, cost_estimated: true };
}
