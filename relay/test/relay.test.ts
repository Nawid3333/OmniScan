import { env, SELF } from "cloudflare:test";
import { describe, expect, it } from "vitest";

const HOOK_SECRET = "test-hook-secret";
const CLIENT_TOKEN = "test-client-token";
const AUTH = { Authorization: `Bearer ${CLIENT_TOKEN}` };

interface EventItem {
  seq: number;
  key: string;
  type: string;
  received_at: number;
  payload: Record<string, unknown>;
}

async function clearEvents(): Promise<void> {
  await SELF.fetch("http://localhost/ack", {
    method: "POST",
    headers: { ...AUTH, "Content-Type": "application/json" },
    body: JSON.stringify({ upto: Number.MAX_SAFE_INTEGER }),
  });
}

const examplePayload = {
  type: "extraction_done" as const,
  data: {
    id: "evt-00000000-0000-0000-0000-000000000001",
    status: "done",
    url: "https://example.com/page",
    images: [
      {
        id: "img-00000000-0000-0000-0000-000000000001",
        url: "https://example.com/a.jpg",
      },
    ],
    created_at: "2026-09-18T00:00:00.000000Z",
    project_id: "prj-00000000-0000-0000-0000-000000000001",
  },
};

describe("relay HTTP endpoints", () => {
  it("GET /health returns 200 ok", async () => {
    const res = await SELF.fetch("http://localhost/health");
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("ok");
  });

  it("wrong hook secret returns 404", async () => {
    const res = await SELF.fetch(
      `http://localhost/hooks/extractpics/wrong-secret`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(examplePayload),
      },
    );
    expect(res.status).toBe(404);
  });

  it("right hook secret stores event and returns duplicate:false", async () => {
    const res = await SELF.fetch(
      `http://localhost/hooks/extractpics/${HOOK_SECRET}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(examplePayload),
      },
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      ok: boolean;
      duplicate: boolean;
      seq: number;
    };
    expect(body.ok).toBe(true);
    expect(body.duplicate).toBe(false);
    expect(body.seq).toBeGreaterThan(0);
  });

  it("duplicate POST returns duplicate:true with same seq", async () => {
    await clearEvents();

    const first = await SELF.fetch(
      `http://localhost/hooks/extractpics/${HOOK_SECRET}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(examplePayload),
      },
    );
    const firstBody = (await first.json()) as {
      ok: boolean;
      duplicate: boolean;
      seq: number;
    };

    const second = await SELF.fetch(
      `http://localhost/hooks/extractpics/${HOOK_SECRET}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(examplePayload),
      },
    );
    const secondBody = (await second.json()) as {
      ok: boolean;
      duplicate: boolean;
      seq: number;
    };

    expect(secondBody.ok).toBe(true);
    expect(secondBody.duplicate).toBe(true);
    expect(secondBody.seq).toBe(firstBody.seq);
  });

  it("invalid JSON returns 400", async () => {
    const res = await SELF.fetch(
      `http://localhost/hooks/extractpics/${HOOK_SECRET}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "not-json",
      },
    );
    expect(res.status).toBe(400);
  });

  it("unknown type returns 422", async () => {
    const res = await SELF.fetch(
      `http://localhost/hooks/extractpics/${HOOK_SECRET}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "unknown_type", data: { id: "x" } }),
      },
    );
    expect(res.status).toBe(422);
  });

  it("missing id and batch_id returns 422", async () => {
    const res = await SELF.fetch(
      `http://localhost/hooks/extractpics/${HOOK_SECRET}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "extraction_done", data: {} }),
      },
    );
    expect(res.status).toBe(422);
  });

  it("GET /events rejects missing or wrong bearer", async () => {
    const missing = await SELF.fetch("http://localhost/events");
    expect(missing.status).toBe(401);

    const wrong = await SELF.fetch("http://localhost/events", {
      headers: { Authorization: "Bearer wrong" },
    });
    expect(wrong.status).toBe(401);
  });

  it("GET /events paginates with after and limit", async () => {
    await clearEvents();

    // Store three distinct events with stable ordering.
    const ids = [crypto.randomUUID(), crypto.randomUUID(), crypto.randomUUID()];
    for (const id of ids) {
      const res = await SELF.fetch(
        `http://localhost/hooks/extractpics/${HOOK_SECRET}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            type: "download_done",
            data: { id },
          }),
        },
      );
      expect(res.status).toBe(200);
    }

    const all = await SELF.fetch("http://localhost/events?limit=500", {
      headers: AUTH,
    });
    const allBody = (await all.json()) as { events: EventItem[]; next: number };
    expect(allBody.events.length).toBe(3);
    expect(allBody.next).toBeGreaterThan(0);

    // Querying after the first event returns the remaining two.
    const firstSeq = allBody.events[0].seq;
    const after = firstSeq;
    const page = await SELF.fetch(`http://localhost/events?after=${after}`, {
      headers: AUTH,
    });
    const pageBody = (await page.json()) as { events: EventItem[]; next: number };
    expect(pageBody.events.length).toBe(2);
    expect(pageBody.events[0]).toHaveProperty("seq", allBody.events[1].seq);
    expect(pageBody.next).toBe(allBody.events[2].seq);

    const limited = await SELF.fetch(
      `http://localhost/events?after=${after}&limit=1`,
      { headers: AUTH },
    );
    const limitedBody = (await limited.json()) as {
      events: EventItem[];
      next: number;
    };
    expect(limitedBody.events.length).toBe(1);
    expect(limitedBody.next).toBe(allBody.events[1].seq);
  });

  it("POST /ack deletes events up to seq", async () => {
    await clearEvents();

    // Seed two events.
    const ids = [crypto.randomUUID(), crypto.randomUUID()];
    for (const id of ids) {
      await SELF.fetch(`http://localhost/hooks/extractpics/${HOOK_SECRET}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "download_done",
          data: { id },
        }),
      });
    }

    const before = await SELF.fetch("http://localhost/events?limit=500", {
      headers: AUTH,
    });
    const beforeBody = (await before.json()) as {
      events: unknown[];
      next: number;
    };
    expect(beforeBody.events.length).toBe(2);
    const upto = beforeBody.next;

    const ack = await SELF.fetch("http://localhost/ack", {
      method: "POST",
      headers: { ...AUTH, "Content-Type": "application/json" },
      body: JSON.stringify({ upto }),
    });
    expect(ack.status).toBe(200);
    const ackBody = (await ack.json()) as { deleted: number };
    expect(ackBody.deleted).toBe(2);

    const after = await SELF.fetch(
      `http://localhost/events?after=${upto}&limit=500`,
      { headers: AUTH },
    );
    const afterBody = (await after.json()) as {
      events: unknown[];
      next: number;
    };
    expect(afterBody.events.length).toBe(0);
    expect(afterBody.next).toBe(upto);
  });
});

describe("relay WebSocket", () => {
  it("/ws without bearer returns 401", async () => {
    const res = await SELF.fetch("http://localhost/ws", {
      headers: { Upgrade: "websocket" },
    });
    expect(res.status).toBe(401);
  });

  it("receives stored events after hello and live broadcasts", async () => {
    // Clear the hub's events so the test starts from a known state.
    const hubId = env.HUB.idFromName("default");
    const hub = env.HUB.get(hubId);
    await hub.fetch(
      new Request("http://localhost/ack", {
        method: "POST",
        headers: { ...AUTH, "Content-Type": "application/json" },
        body: JSON.stringify({ upto: Number.MAX_SAFE_INTEGER }),
      }),
    );

    // Seed an event through the hook.
    const seedId = crypto.randomUUID();
    await SELF.fetch(`http://localhost/hooks/extractpics/${HOOK_SECRET}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "extraction_done",
        data: { id: seedId },
      }),
    });

    // Connect via the Durable Object directly (same code path as the worker).
    const wsReq = new Request("http://localhost/ws", {
      headers: {
        ...AUTH,
        Upgrade: "websocket",
      },
    });
    const wsRes = await hub.fetch(wsReq);
    expect(wsRes.status).toBe(101);

    // The DO accepted the server socket and returned the client socket in the response.
    const clientSocket = wsRes.webSocket;
    if (!clientSocket) {
      throw new Error("Expected client WebSocket in response");
    }
    clientSocket.accept();

    const messages: unknown[] = [];
    clientSocket.addEventListener("message", (event) => {
      messages.push(JSON.parse(event.data as string));
    });

    // Send hello to trigger replay of stored events.
    clientSocket.send(JSON.stringify({ type: "hello", after: 0 }));

    // Wait for replayed event.
    await waitFor(() => messages.length > 0, 1000);
    expect(messages.length).toBe(1);
    const replayed = messages[0] as { seq: number; type: string };
    expect(replayed.type).toBe("extraction_done");

    // Post a new hook and expect a live broadcast.
    const liveId = crypto.randomUUID();
    await SELF.fetch(`http://localhost/hooks/extractpics/${HOOK_SECRET}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "extraction_done",
        data: { id: liveId },
      }),
    });

    await waitFor(() => messages.length > 1, 1000);
    expect(messages.length).toBe(2);
    const live = messages[1] as { seq: number; type: string };
    expect(live.type).toBe("extraction_done");
    expect(live.seq).toBeGreaterThan(replayed.seq);

    clientSocket.close();
  });
});

function waitFor(predicate: () => boolean, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const check = () => {
      if (predicate()) {
        resolve();
      } else if (Date.now() >= deadline) {
        reject(new Error("Timed out waiting for condition"));
      } else {
        setTimeout(check, 10);
      }
    };
    check();
  });
}
