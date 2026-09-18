export interface Env {
  HUB: DurableObjectNamespace;
  HOOK_SECRET: string;
  CLIENT_TOKEN: string;
}

export const HOOK_TYPES = [
  "extraction_done",
  "extraction_batch_done",
  "download_done",
] as const;

export type HookType = (typeof HOOK_TYPES)[number];

export interface HookBody {
  type: HookType;
  data: {
    id?: string;
    batch_id?: string;
    [key: string]: unknown;
  };
}

export interface EventRecord {
  seq: number;
  key: string;
  type: HookType;
  received_at: number;
  payload: HookBody;
}

export interface StoreResult {
  seq: number;
  duplicate: boolean;
}

export interface WebSocketMessage {
  type: "hello";
  after?: number;
}
