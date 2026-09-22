import { handleAdmin } from "./api/admin";
import { handleHealth } from "./api/health";
import { handleCache } from "./api/cache";
import { handleCacheHealth, handleVectorHealth, handleVectorReindex } from "./api/debug";
import { handleChatCompletions } from "./api/chatCompletions";
import { handleGuideDogChatCompletions } from "./api/guideDog";
import { handleIngestMessagesApi, handleMemories, handleSearchMemoriesApi } from "./api/memories";
import { handleMcp } from "./api/mcp";
import { handleModels } from "./api/models";
import { handleBillingBalances } from "./api/billing";
import { handleDeleteConversation } from "./api/conversations";
import { handleReplySelection } from "./api/replySelection";
import { handleRelayModels } from "./api/relay";
import { handleSourceMessageSync } from "./api/sourceMessages";
import { runDailyMemoryDigest } from "./memory/dailyDigest";
import { runMemoryRetention } from "./memory/retention";
import { handleQueueMessage } from "./queue/consumer";
import type { Env, QueueMessage } from "./types";
import { openAiError } from "./utils/json";

function getDailyDigestNamespace(env: Env): string {
  return env.DREAM_NAMESPACE?.trim() || "default";
}

function getDailyDigestMaxRuns(env: Env): number {
  const parsed = Number(env.DREAM_MAX_RUNS || env.DAILY_DIGEST_MAX_RUNS || 10);
  if (!Number.isFinite(parsed)) return 10;
  return Math.min(Math.max(Math.floor(parsed), 1), 10);
}

async function runDailyMemoryDigestBatches(
  env: Env,
  namespace: string,
  executionTime: string
): Promise<unknown[]> {
  const results: unknown[] = [];
  const maxRuns = getDailyDigestMaxRuns(env);

  for (let i = 0; i < maxRuns; i += 1) {
    const result = await runDailyMemoryDigest(env, namespace, { executionTime });
    results.push(result);
    if (!result.ran || !result.stats?.hasMore) break;
  }

  return results;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && (url.pathname === "/admin" || url.pathname === "/memory-admin")) {
      return handleAdmin();
    }

    if (request.method === "GET" && url.pathname === "/health") {
      return handleHealth(env);
    }

    if (request.method === "GET" && url.pathname === "/v1/models") {
      return handleModels(request, env);
    }

    if (request.method === "POST" && url.pathname === "/v1/relay/models") {
      return handleRelayModels(request, env);
    }

    if (request.method === "GET" && url.pathname === "/v1/billing/balances") {
      return handleBillingBalances(request, env);
    }

    if (["GET", "POST"].includes(request.method) && /^\/v1\/conversations\/[^/]+\/reply-selection$/.test(url.pathname)) {
      return handleReplySelection(request, env);
    }

    if (request.method === "DELETE" && url.pathname.startsWith("/v1/conversations/")) {
      return handleDeleteConversation(request, env);
    }

    if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
      return handleChatCompletions(request, env, ctx);
    }

    if (
      request.method === "POST" &&
      (url.pathname === "/v1/guide-dog/chat/completions" || url.pathname === "/guide-dog/v1/chat/completions")
    ) {
      return handleGuideDogChatCompletions(request, env);
    }

    if (url.pathname === "/mcp" || url.pathname === "/memory-mcp") {
      return handleMcp(request, env, ctx);
    }

    if (url.pathname.startsWith("/v1/memories")) {
      return handleMemories(request, env, ctx);
    }

    if (request.method === "POST" && url.pathname === "/v1/memory/source-messages/sync") {
      return handleSourceMessageSync(request, env);
    }

    if (url.pathname === "/v1/memory" || url.pathname.startsWith("/v1/memory/")) {
      return handleMemories(request, env, ctx);
    }

    if (
      request.method === "POST" &&
      (url.pathname === "/v1/ingest/messages" || url.pathname === "/v1/messages/ingest")
    ) {
      return handleIngestMessagesApi(request, env, ctx);
    }

    if (request.method === "POST" && url.pathname === "/v1/search/memories") {
      return handleSearchMemoriesApi(request, env);
    }

    if (url.pathname.startsWith("/v1/cache/")) {
      return handleCache(request, env);
    }

    if (request.method === "GET" && url.pathname === "/v1/debug/cache_health") {
      return handleCacheHealth(request, env);
    }

    if (request.method === "GET" && url.pathname === "/v1/debug/vector_health") {
      return handleVectorHealth(request, env);
    }

    if (request.method === "POST" && url.pathname === "/v1/debug/vector_reindex") {
      return handleVectorReindex(request, env);
    }

    return openAiError("Not found", 404);
  },

  async queue(batch: MessageBatch<QueueMessage>, env: Env): Promise<void> {
    for (const message of batch.messages) {
      try {
        await handleQueueMessage(message.body, env);
        message.ack();
      } catch (error) {
        console.error("queue message failed", error);
        message.retry();
      }
    }
  },

  async scheduled(_controller: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    const namespace = getDailyDigestNamespace(env);
    const executionTime = new Date().toISOString();
    ctx.waitUntil(
      Promise.all([
        runDailyMemoryDigestBatches(env, namespace, executionTime),
        runMemoryRetention(env, namespace)
      ]).then(([digest, retention]) => {
        console.log("scheduled daily memory maintenance", { namespace, digest, retention });
      })
    );
  }
};
