export interface ReplyVariant {
  turnId: string;
  variantId: string;
}

const ID = /^[a-zA-Z0-9_-]{1,100}$/;

export function parseReplyVariant(value: unknown): ReplyVariant | undefined {
  if (value == null) return undefined;
  const input = value as Partial<ReplyVariant>;
  if (!input || typeof input.turnId !== "string" || typeof input.variantId !== "string"
      || !ID.test(input.turnId) || !ID.test(input.variantId)) {
    throw new Error("Invalid reply variant");
  }
  return { turnId: input.turnId!, variantId: input.variantId! };
}
