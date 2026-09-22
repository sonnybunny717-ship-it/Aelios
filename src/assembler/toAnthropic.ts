/**
 * Pure conversion: AssembledPrompt → Anthropic wire format types.
 *
 * These helpers do NOT call any adapter, DB, or external service.
 * The existing anthropicAdapter.ts is untouched; adapters will import
 * these functions in P1.3 integration (a later step).
 *
 * Determinism: given the same AssembledPrompt, output is bit-for-bit identical.
 */

import type { AssembledPrompt, SystemBlock } from "./types";

// ---------------------------------------------------------------------------
// Anthropic wire types (subset needed for system + messages)
// ---------------------------------------------------------------------------

export interface AnthropicTextBlock {
  type: "text";
  text: string;
  cache_control?: {
    type: "ephemeral";
    ttl?: "5m" | "1h";
  };
}

export interface AnthropicImageBlock {
  type: "image";
  source:
    | { type: "url"; url: string }
    | { type: "base64"; media_type: "image/jpeg" | "image/png" | "image/gif" | "image/webp"; data: string };
  cache_control?: {
    type: "ephemeral";
    ttl?: "5m" | "1h";
  };
}

export type AnthropicContentBlock = AnthropicTextBlock | AnthropicImageBlock;

export interface AnthropicWireMessage {
  role: "user" | "assistant";
  content: AnthropicContentBlock[];
}

// ---------------------------------------------------------------------------
// System blocks → AnthropicTextBlock[]
// ---------------------------------------------------------------------------

/**
 * Convert AssembledPrompt.system_blocks to Anthropic system format.
 * Preserves cache_control exactly as set by the assembler.
 */
export function assembledToAnthropicSystem(
  systemBlocks: SystemBlock[]
): AnthropicTextBlock[] {
  return systemBlocks.map((block) => {
    const out: AnthropicTextBlock = { type: "text", text: block.text };
    if (block.cache_control) {
      out.cache_control = {
        type: "ephemeral",
        ...(block.cache_control.ttl ? { ttl: block.cache_control.ttl } : {}),
      };
    }
    return out;
  });
}

// ---------------------------------------------------------------------------
// Messages → AnthropicMessage[]
// ---------------------------------------------------------------------------

/**
 * Convert AssembledPrompt.messages to Anthropic message format.
 *
 * Converts OpenAI text/image_url blocks to Anthropic text/image blocks.
 * Unknown structured blocks are preserved as JSON text instead of being dropped.
 */
export function assembledToAnthropicMessages(
  messages: AssembledPrompt["messages"]
): AnthropicWireMessage[] {
  const result: AnthropicWireMessage[] = [];

  for (const msg of messages) {
    const role = msg.role;
    const blocks = openAIContentToAnthropicBlocks(msg.content);

    const prev = result[result.length - 1];
    if (prev?.role === role) {
      prev.content.push(...blocks);
      continue;
    }

    result.push({ role, content: blocks });
  }

  if (result.length === 0) {
    result.push({ role: "user", content: [{ type: "text", text: "" }] });
  }

  return result;
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function imageBlockFromUrl(url: string): AnthropicImageBlock | null {
  const dataMatch = url.match(/^data:(image\/(?:jpeg|png|gif|webp));base64,([\s\S]+)$/i);
  if (dataMatch) {
    return {
      type: "image",
      source: {
        type: "base64",
        media_type: dataMatch[1].toLowerCase() as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
        data: dataMatch[2],
      },
    };
  }
  if (/^https?:\/\//i.test(url)) {
    return { type: "image", source: { type: "url", url } };
  }
  return null;
}

export function openAIContentToAnthropicBlocks(
  content: string | unknown[] | null
): AnthropicContentBlock[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (content == null) return [{ type: "text", text: "" }];

  const blocks: AnthropicContentBlock[] = [];
  for (const part of content) {
    if (!isRecord(part)) {
      blocks.push({ type: "text", text: JSON.stringify(part) });
      continue;
    }
    if (part.type === "text" && typeof part.text === "string") {
      blocks.push({ type: "text", text: part.text });
      continue;
    }
    if (part.type === "image_url" || part.type === "input_image") {
      const rawImage = part.image_url;
      const url = typeof rawImage === "string"
        ? rawImage
        : isRecord(rawImage) && typeof rawImage.url === "string"
          ? rawImage.url
          : typeof part.url === "string"
            ? part.url
            : "";
      const image = imageBlockFromUrl(url);
      if (image) {
        blocks.push(image);
        continue;
      }
    }
    blocks.push({ type: "text", text: JSON.stringify(part) });
  }
  return blocks.length ? blocks : [{ type: "text", text: "" }];
}
