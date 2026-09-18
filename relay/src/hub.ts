import { DurableObject } from "cloudflare:workers";

type DurableObjectState = ConstructorParameters<typeof DurableObject>[0];
import {
  HOOK_TYPES,
  type Env,
  type EventRecord,
  type HookBody,
  type StoreResult,
  type WebSocketMessage,
} from "./types.js";

const MAX_BODY_BYTES = 5 * 1024 * 1024;
const DEFAULT_LIMIT = 100;
const MAX_LIMIT = 500;
const RETENTION_DAYS = 14;
const ALARM_INTERVAL_MS = 24 * 60 * 60 * 1000;

import type { SqlStorageValue } from "@cloudflare/workers-types";

interface EventRow extends Record<string, SqlStorageValue> {
  seq: number;
  key: string;
  type: string;
  received_at: number;
  payload: string;
}

export class EventHub extends DurableObject<Env> {
  private sql: SqlStorage;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.sql = ctx.storage.sql;
    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        type TEXT NOT NULL,
        received_at INTEGER NOT NULL,
        payload TEXT NOT NULL
      );
    `);
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (method === "GET" && path === "/health") {
      return new Response("ok", { status: 200 });
    }

    if (method === "POST" && path.startsWith("/hooks/extractpics/")) {
      return this.handleHook(request, path);
    }

    if (method === "GET" && path === "/events") {
      return this.handleEvents(url);
    }

    if (method === "POST" && path === "/ack") {
      return this.handleAck(request);
    }

    if (method === "GET" && path === "/ws") {
      return this.handleWebSocket(request);
    }

    return new Response("Not found", { status: 404 });
  }

  private async handleHook(request: Request, path: string): Promise<Response> {
    const secret = path.slice("/hooks/extractpics/".length);
    const expected = this.env.HOOK_SECRET;

    if (!constantTimeEqual(secret, expected)) {
      return new Response("Not found", { status: 404 });
    }

    const contentLength = request.headers.get("content-length");
    if (contentLength && parseInt(contentLength, 10) > MAX_BODY_BYTES) {
      return new Response("Payload too large", { status: 413 });
    }

    const body = await request.text();
    if (body.length > MAX_BODY_BYTES) {
      return new Response("Payload too large", { status: 413 });
    }

    let payload: HookBody;
    try {
      payload = JSON.parse(body) as HookBody;
    } catch {
      return new Response("Invalid JSON", { status: 400 });
    }

    if (!HOOK_TYPES.includes(payload.type)) {
      return new Response("Unsupported type", { status: 422 });
    }

    const id = payload.data?.id;
    const batchId = payload.data?.batch_id;
    const rawId = id ?? batchId;
    if (typeof rawId !== "string" || rawId.length === 0) {
      return new Response("Missing id or batch_id", { status: 422 });
    }

    const key = `${payload.type}:${rawId}`;
    const receivedAt = Date.now();

    const result = (await this.ctx.blockConcurrencyWhile(async () => {
      const existing = this.sql
        .exec<{ seq: number }>("SELECT seq FROM events WHERE key = ?", key)
        .toArray()[0];
      if (existing) {
        return { seq: existing.seq, duplicate: true } as StoreResult;
      }

      this.sql.exec(
        "INSERT INTO events (key, type, received_at, payload) VALUES (?, ?, ?, ?)",
        key,
        payload.type,
        receivedAt,
        JSON.stringify(payload),
      );

      const row = this.sql
        .exec<{ seq: number }>("SELECT last_insert_rowid() AS seq")
        .toArray()[0];
      if (!row) {
        throw new Error("Failed to read inserted event");
      }

      await this.ensureAlarm();

      return { seq: row.seq, duplicate: false } as StoreResult;
    })) as StoreResult;

    if (!result.duplicate) {
      const event: EventRecord = {
        seq: result.seq,
        key,
        type: payload.type,
        received_at: receivedAt,
        payload,
      };
      this.broadcast(event);
    }

    return Response.json({ ok: true, duplicate: result.duplicate, seq: result.seq });
  }

  private handleEvents(url: URL): Response {
    const afterParam = url.searchParams.get("after");
    const after = afterParam ? parseInt(afterParam, 10) : 0;
    if (Number.isNaN(after)) {
      return new Response("Invalid after", { status: 400 });
    }

    const limitParam = url.searchParams.get("limit");
    let limit = limitParam ? parseInt(limitParam, 10) : DEFAULT_LIMIT;
    if (Number.isNaN(limit)) {
      limit = DEFAULT_LIMIT;
    }
    limit = Math.max(1, Math.min(limit, MAX_LIMIT));

    const rows = this.sql
      .exec<EventRow>(
        `SELECT seq, key, type, received_at, payload FROM events
         WHERE seq > ? ORDER BY seq ASC LIMIT ?`,
        after,
        limit,
      )
      .toArray();

    const events = rows.map((row) => ({
      seq: row.seq,
      key: row.key,
      type: row.type,
      received_at: row.received_at,
      payload: JSON.parse(row.payload) as HookBody,
    }));
    const next = rows.length > 0 ? rows[rows.length - 1].seq : after;

    return Response.json({ events, next });
  }

  private async handleAck(request: Request): Promise<Response> {
    let body: { upto?: number };
    try {
      body = (await request.json()) as { upto?: number };
    } catch {
      return new Response("Invalid JSON", { status: 400 });
    }

    const upto = body.upto;
    if (typeof upto !== "number" || Number.isNaN(upto)) {
      return new Response("Invalid upto", { status: 400 });
    }

    const result = this.sql.exec<{ seq: number }>(
      "DELETE FROM events WHERE seq <= ? RETURNING seq",
      upto,
    );
    let deleted = 0;
    for (const _row of result) {
      deleted += 1;
    }

    return Response.json({ deleted });
  }

  private handleWebSocket(request: Request): Response {
    const upgradeHeader = request.headers.get("Upgrade");
    if (upgradeHeader !== "websocket") {
      return new Response("Upgrade required", { status: 426 });
    }

    const [client, server] = Object.values(new WebSocketPair());
    this.ctx.acceptWebSocket(server);

    return new Response(null, { status: 101, webSocket: client });
  }

  async webSocketMessage(
    ws: WebSocket,
    message: string | ArrayBuffer,
  ): Promise<void> {
    if (typeof message !== "string") return;

    let msg: WebSocketMessage;
    try {
      msg = JSON.parse(message) as WebSocketMessage;
    } catch {
      return;
    }

    if (msg.type === "hello" && typeof msg.after === "number") {
      const rows = this.sql
        .exec<EventRow>(
          `SELECT seq, key, type, received_at, payload FROM events
           WHERE seq > ? ORDER BY seq ASC`,
          msg.after,
        )
        .toArray();

      for (const row of rows) {
        const event: EventRecord = {
          seq: row.seq,
          key: row.key,
          type: row.type as HookBody["type"],
          received_at: row.received_at,
          payload: JSON.parse(row.payload) as HookBody,
        };
        ws.send(JSON.stringify(event));
      }
    }
  }

  private broadcast(event: EventRecord): void {
    const message = JSON.stringify(event);
    for (const ws of this.ctx.getWebSockets()) {
      ws.send(message);
    }
  }

  async alarm(): Promise<void> {
    const cutoff = Date.now() - RETENTION_DAYS * 24 * 60 * 60 * 1000;
    this.sql.exec("DELETE FROM events WHERE received_at < ?", cutoff);
    await this.ensureAlarm();
  }

  private async ensureAlarm(): Promise<void> {
    const current = await this.ctx.storage.getAlarm();
    if (current === null) {
      await this.ctx.storage.setAlarm(Date.now() + ALARM_INTERVAL_MS);
    }
  }
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let result = 0;
  for (let i = 0; i < a.length; i += 1) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
